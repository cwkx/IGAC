function phi = cpuProcess(I, params)

    % Get params
    sigma       = params{find(strcmp(params(:,1), '-r')), 2};
    timestep    = 0.1;
    mu          = 1;
    nu          = params{find(strcmp(params(:,1), '-s')), 2};
    alf         = 30;
    lambda1     = 1 + max(0, -params{find(strcmp(params(:,1), '-g')), 2});
    lambda2     = 1 + max(0,  params{find(strcmp(params(:,1), '-g')), 2});
    maxiter     = params{find(strcmp(params(:,1), '--maxiter')), 2};
    cx          = params{find(strcmp(params(:,1), '--cx')), 2};
    cy          = params{find(strcmp(params(:,1), '--cy')), 2};
    cz          = params{find(strcmp(params(:,1), '--cz')), 2};
    cr          = params{find(strcmp(params(:,1), '--cr')), 2};
    
    % Normalize and apply awgn
    I = (I - min(I(:)))/(max(I(:))-min(I(:)));
    I = imnoise(I, 'gaussian', params{find(strcmp(params(:,1), '--awgn')), 2});
    
    [h, w, d] = size(I);
    [xx, yy, zz] = meshgrid(1:w,1:h,1:d);
    if (size(I,3) > 1)
        phi = (sqrt(((xx - cx).^2 + (yy - cy).^2 + (zz - cz).^2)) - cr);
        phi = sign(phi).*2;
        KI = gauss3filter(I, sigma);
        KI2 = gauss3filter(I.^2, sigma);
    else
        phi = (sqrt(((xx - cx).^2 + (yy - cy).^2 )) - cr);
        phi = sign(phi).*2;
        KI = imgaussfilt(I, sigma);
        KI2 = imgaussfilt(I.^2, sigma);
    end

    iter = 0;
    while true
        
        tic
        % First Filter
        if (size(I,3) > 1)
            phi = neumannBound3D(phi);
            H = heavisideFunction(phi);
            K = curvatureCentral3D(phi);
            GT(:,:,:,1) = (H.*I);
            GT(:,:,:,2) = H;
            GT(:,:,:,3) = I.^2.*H;
            GT = gauss3filter(GT, sigma);
            GIH  = GT(:,:,:,1); % zeroing the image inside the contour, then blurring
            GH   = GT(:,:,:,2); % blurring the Heaviside function
            GI2H = GT(:,:,:,3); % local sum of squared intensity outside contour
        else
            phi = neumannBound(phi);
            H = heavisideFunction(phi);
            K = curvatureCentral(phi);
            GT(:,:,1) = (H.*I);
            GT(:,:,2) = H;
            GT(:,:,3) = I.^2.*H;
            GIH  = imgaussfilt(GT(:,:,1), sigma); % zeroing the image inside the contour, then blurring
            GH   = imgaussfilt(GT(:,:,2), sigma); % blurring the Heaviside function
            GI2H = imgaussfilt(GT(:,:,3), sigma); % local sum of squared intensity outside contour
        end

        % Second Filter Prep
        u1= GIH./(GH);
        u2 = (KI - GIH)./(1 - GH);
        sigma1 = (GI2H ./ GH) - u1.^2;
        sigma2 = ((KI2-GI2H)./(1.0-GH))-u2.^2;
        sigma1 = sigma1 + 1e-10;
        sigma2 = sigma2 + 1e-10;

        % Second Filter
        if (size(I,3) > 1)
            A(:,:,:,1) = lambda1.*log(sqrt(sigma1)) - lambda2.*log(sqrt(sigma2)) +lambda1.*u1.^2./(2.*sigma1) - lambda2.*u2.^2./(2.*sigma2);
            A(:,:,:,2) = lambda2.*u2./sigma2 - lambda1.*u1./sigma1;
            A(:,:,:,3) = lambda1.*1./(2.*sigma1) - lambda2.*1./(2.*sigma2);
            A = gauss3filter(A, sigma);
            localForce = (lambda1 - lambda2).*log(sqrt(2*pi)); % constant stuff
            localForce = localForce + A(:,:,:,1);
            localForce = localForce + I.*A(:,:,:,2);
            localForce = localForce + I.^2 .* A(:,:,:,3);
        else
            A(:,:,1) = lambda1.*log(sqrt(sigma1)) - lambda2.*log(sqrt(sigma2)) +lambda1.*u1.^2./(2.*sigma1) - lambda2.*u2.^2./(2.*sigma2);
            A(:,:,2) = lambda2.*u2./sigma2 - lambda1.*u1./sigma1;
            A(:,:,3) = lambda1.*1./(2.*sigma1) - lambda2.*1./(2.*sigma2);
            localForce = (lambda1 - lambda2).*log(sqrt(2*pi)); % constant stuff
            localForce = localForce + imgaussfilt(A(:,:,1), sigma);
            localForce = localForce + I.*imgaussfilt(A(:,:,2), sigma);
            localForce = localForce + I.^2 .* imgaussfilt(A(:,:,3), sigma);
        end

        % Update
        Delta = diracFunction(phi);
        A = -alf.*Delta.*localForce;
        P=mu*(4*del2(phi) - K);
        L=nu.*Delta.*K;
        phi = phi+timestep*(L+P+A);

        % Display
        if (mod(iter,10)==0)
            if (size(I,3) > 1)
                figure(1), imagesc(phi(:,:,floor(size(I,3)/2)));
                colormap(gray),axis off;axis equal,title(num2str(iter))
                hold on, contour(phi(:,:,floor(size(I,3)/2)),[0 0],'r','linewidth',1); hold off; drawnow;
            else
                figure(1),
                imshow(mat2gray(phi)),colormap(gray),axis off;axis equal,title(num2str(iter))
                hold on, contour(phi,[0 0],'r','linewidth',1); hold off; drawnow;
            end
        end

        iter = iter+1
        if (iter > maxiter && maxiter > 0)
            close;
            return;
        end
    end
end