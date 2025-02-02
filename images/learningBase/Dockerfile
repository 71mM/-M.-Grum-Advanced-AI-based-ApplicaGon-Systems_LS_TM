# Use busybox as base image
FROM busybox

# Set the working directory
WORKDIR /tmp/learningBase

# Copy data files into container
COPY train /tmp/learningBase/train/
COPY validation /tmp/learningBase/validation/
COPY Readme.md /tmp/learningBase/

# Set default command
CMD ["sh"]